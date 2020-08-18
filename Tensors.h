namespace SimulationUtilities{
	namespace
	{
		/////------------------------------------Tensor expressions (heart of tensor arithmetic)-------------------------------------\\\\\

		//type to signify subtraction and division version addition and multiplication
		template<bool isInverse>
		struct InverseType{static constexpr bool value = isInverse;};

		//represents the type of any given expression of indexed tensors.
		//comes in the form of single (stores a single indexed tensor), 
		//product (stores two sub-expressions being multiplied),
		//sum (stores two sub-expressions being added or subtracted),
		//and scalar product (stores sub-expression and multiplier).
		template<char ExpressionIdentifier, size_t dimensions, typename... Ts>
		struct Expression;

		//the single version of Expression.
		//must be able to handle arbitrary trace (RepeatIs... represents the trace indices),
		//and non-traced assignments (=, +=, -=, *=)
		template<size_t rank, size_t dimensions, typename T, typename... FreeIndices, typename... Is, typename... RepeatIs>
		struct Expression<'s', dimensions, T, IndexPackType<FreeIndices...>, IndexedTensor<rank, dimensions, T, Is...>, IndexPackType<RepeatIs...>>
		{
			T* data;
			size_t activeIndexingLocations[rank];

			Expression(T* initData)
			:
				data(initData)
			{
				for (size_t i = 0; i < rank; ++i)
				{
					activeIndexingLocations[i] = 0;
				}
			}

			Expression(const Expression<'s', dimensions, T, IndexPackType<FreeIndices...>,
				IndexedTensor<rank, dimensions, T, Is...>, IndexPackType<RepeatIs...>>& other)
			:
				data(other.data)
			{
				for (size_t i = 0; i < rank; ++i)
				{
					activeIndexingLocations[i] = other.activeIndexingLocations[i];
				}
			}

			template<typename NextMetaIndex, typename... OtherMetaIs>
			struct IndexBuilder
			{
				static inline void buildIndex(size_t& indexThusFar, const size_t* const& indexingLocations)
				{
					constexpr size_t indexPosition = rank - sizeof...(OtherMetaIs) - 1;
					indexThusFar += indexingLocations[indexPosition];
					indexThusFar *= dimensions;
					IndexBuilder<OtherMetaIs...>::buildIndex(indexThusFar, indexingLocations);
				}
			};

			template<typename LastMetaIndex>
			struct IndexBuilder<LastMetaIndex>
			{
				static inline void buildIndex(size_t& indexThusFar, const size_t* const& indexingLocations)
				{
					indexThusFar += indexingLocations[rank - 1];
				}
			};

			template<typename NextMetaIndex, typename... OtherMetaIs>
			struct GetValue_Helper
			{
				static inline void getValue(T& valueThusFar, T* const& data, size_t* const& activeIndexingLocations)
				{
					static constexpr size_t i0 = Template_Locate_Nth_Key_Type<0, NextMetaIndex, Is...>::value;
					static constexpr size_t i1 = Template_Locate_Nth_Key_Type<1, NextMetaIndex, Is...>::value;
					for (size_t i = 0; i < dimensions; ++i)
					{
						activeIndexingLocations[i0] = i;
						activeIndexingLocations[i1] = i;
						GetValue_Helper<OtherMetaIs...>::getValue(valueThusFar, data, activeIndexingLocations);
					}
				}
			};

			template<typename LastMetaIndex>
			struct GetValue_Helper<LastMetaIndex>
			{
				static inline void getValue(T& valueThusFar, T* const& data, size_t* const& activeIndexingLocations)
				{
					static constexpr size_t i0 = Template_Locate_Nth_Key_Type<0, LastMetaIndex, Is...>::value;
					static constexpr size_t i1 = Template_Locate_Nth_Key_Type<1, LastMetaIndex, Is...>::value;
					for (size_t i = 0; i < dimensions; ++i)
					{
						activeIndexingLocations[i0] = i;
						activeIndexingLocations[i1] = i;
						size_t index = 0;
						IndexBuilder<Is...>::buildIndex(index, activeIndexingLocations);
						valueThusFar += data[index];
					}
				}
			};

			//generic expression requirements

			inline T getValue()
			{
				if constexpr (sizeof...(RepeatIs) == 0)
				{
					if constexpr (sizeof...(Is) == 0)
					{
						return data[0];
					}
					else
					{
						size_t index = 0;
						IndexBuilder<Is...>::buildIndex(index, activeIndexingLocations);
						return data[index];
					}
				}
				else
				{
					T output = T();
					GetValue_Helper<RepeatIs...>::getValue(output, data, activeIndexingLocations);
					return output;
				}
			}

			template<typename Index>
			inline void setIndex(const size_t& indexValue)
			{
				if constexpr (Template_Key_In_Pack<Index, Is...>::value)
				{
					activeIndexingLocations[Template_Locate_Key_Type<Index, Is...>::value] = indexValue;
				}
			}

			//single expression requirements. needs to handle = operator and similar things

			typedef Expression<'s', dimensions, T, IndexPackType<FreeIndices...>, IndexedTensor<rank, dimensions, T, Is...>, IndexPackType<RepeatIs...>> SelfType;

			template<typename NextMetaIndex, typename... OtherMetaIs>
			struct Equal_Helper
			{
				template<char ID, typename... OtherIs>
				static inline void setEqual(T* const& data, Expression<ID, dimensions, T, OtherIs...>& other, size_t index)
				{
					constexpr size_t incr = Template_Power<dimensions, sizeof...(OtherMetaIs)>::value;
					for (size_t i = 0; i < dimensions; ++i)
					{
						other.template setIndex<NextMetaIndex>(i);
						Equal_Helper<OtherMetaIs...>::setEqual(data, other, index + i * incr);
					}
				}
			};

			template<typename LastMetaIndex>
			struct Equal_Helper<LastMetaIndex>
			{
				template<char ID, typename... OtherIs>
				static inline void setEqual(T* const& data, Expression<ID, dimensions, T, OtherIs...>& other, size_t index)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						other.template setIndex<LastMetaIndex>(i);
						data[index + i] = other.getValue();
					}
				}
			};

			template<char ID, typename... OtherIs>
			SelfType& operator=(Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, OtherIs...>&& other)
			{
				static_assert(sizeof...(RepeatIs) == 0);
				if constexpr (sizeof...(Is) == 0)
				{
					data[0] = other.getValue();
				}
				else
				{
					Equal_Helper<Is...>::setEqual(data, other, 0);
				}
				return *this;
			}

			SelfType& operator=(SelfType&& other)
			{
				for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
				{
					data[i] = other.data[i];
				}
				return *this;
			}

			template<typename NextMetaIndex, typename... OtherMetaIs>
			struct Plus_Helper
			{
				template<char ID, typename... OtherIs>
				static inline void add(T* const& data, Expression<ID, dimensions, T, OtherIs...>& other, size_t index)
				{
					constexpr size_t incr = Template_Power<dimensions, sizeof...(OtherMetaIs)>::value;
					for (size_t i = 0; i < dimensions; ++i)
					{
						other.template setIndex<NextMetaIndex>(i);
						Plus_Helper<OtherMetaIs...>::add(data, other, index + i * incr);
					}
				}
			};

			template<typename LastMetaIndex>
			struct Plus_Helper<LastMetaIndex>
			{
				template<char ID, typename... OtherIs>
				static inline void add(T* const& data, Expression<ID, dimensions, T, OtherIs...>& other, size_t index)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						other.template setIndex<LastMetaIndex>(i);
						data[index + i] += other.getValue();
					}
				}
			};

			template<char ID, typename... OtherIs>
			SelfType& operator+=(Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, OtherIs...>&& other)
			{
				static_assert(sizeof...(RepeatIs) == 0);
				if constexpr (sizeof...(Is) == 0)
				{
					data[0] += other.getValue();
				}
				else
				{
					Plus_Helper<Is...>::add(data, other, 0);
				}
				return *this;
			}

			template<typename NextMetaIndex, typename... OtherMetaIs>
			struct Minus_Helper
			{
				template<char ID, typename... OtherIs>
				static inline void subtract(T* const& data, Expression<ID, dimensions, T, OtherIs...>& other, size_t index)
				{
					constexpr size_t incr = Template_Power<dimensions, sizeof...(OtherMetaIs)>::value;
					for (size_t i = 0; i < dimensions; ++i)
					{
						other.template setIndex<NextMetaIndex>(i);
						Minus_Helper<OtherMetaIs...>::subtract(data, other, index + i * incr);
					}
				}
			};

			template<typename LastMetaIndex>
			struct Minus_Helper<LastMetaIndex>
			{
				template<char ID, typename... OtherIs>
				static inline void subtract(T* const& data, Expression<ID, dimensions, T, OtherIs...>& other, size_t index)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						other.template setIndex<LastMetaIndex>(i);
						data[index + i] -= other.getValue();
					}
				}
			};

			template<char ID, typename... OtherIs>
			SelfType& operator-=(Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, OtherIs...>&& other)
			{
				static_assert(sizeof...(RepeatIs) == 0);
				if constexpr (sizeof...(Is) == 0)
				{
					data[0] -= other.getValue();
				}
				else
				{
					Minus_Helper<Is...>::subtract(data, other, 0);
				}
				return *this;
			}
		};

		//the product version of Expression.
		//must be able to handle arbitrary index contraction (ContractionIndices... represents these indices)
		template<size_t dimensions, typename T, typename... FreeIndices,
			char ID1, char ID2, typename... Is1, typename... Is2, typename... ContractionIndices, typename Inverter>
		struct Expression<'m', dimensions, T, IndexPackType<FreeIndices...>,
			Expression<ID1, dimensions, T, Is1...>, Expression<ID2, dimensions, T, Is2...>, IndexPackType<ContractionIndices...>, Inverter>
		{
			Expression<ID1, dimensions, T, Is1...> val1;
			Expression<ID2, dimensions, T, Is2...> val2;

			template<typename NextMetaIndex, typename... OtherMetaIs>
			struct GetValue_Helper
			{
				static inline void getValue(T& valueThusFar, Expression<ID1, dimensions, T, Is1...>& val1,
					Expression<ID2, dimensions, T, Is2...>& val2)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						val1.template setIndex<NextMetaIndex>(i);
						val2.template setIndex<NextMetaIndex>(i);
						GetValue_Helper<OtherMetaIs...>::getValue(valueThusFar, val1, val2);
					}
				}
			};

			template<typename LastMetaIndex>
			struct GetValue_Helper<LastMetaIndex>
			{
				static inline void getValue(T& valueThusFar, Expression<ID1, dimensions, T, Is1...>& val1,
					Expression<ID2, dimensions, T, Is2...>& val2)
				{
					for (size_t i = 0; i < dimensions; ++i)
					{
						val1.template setIndex<LastMetaIndex>(i);
						val2.template setIndex<LastMetaIndex>(i);
						if constexpr (Inverter::value)
						{
							valueThusFar += val1.getValue() / val2.getValue();
						}
						else
						{
							valueThusFar += val1.getValue() * val2.getValue();
						}
					}
				}
			};

			//generic expression requirements

			inline T getValue()
			{
				if constexpr (sizeof...(ContractionIndices) == 0)
				{
					if constexpr (Inverter::value)
					{
						return val1.getValue() / val2.getValue();
					}
					else
					{
						return val1.getValue() * val2.getValue();
					}
				}
				else
				{
					T output;
					GetValue_Helper<ContractionIndices...>::getValue(output, val1, val2);
					return output;
				}
			}

			template<typename Index>
			inline void setIndex(const size_t& indexValue)
			{
				val1.template setIndex<Index>(indexValue);
				val2.template setIndex<Index>(indexValue);
			}
		};

		//the sum version of Expression.
		template<size_t dimensions, typename T, typename... FreeIndices,
			char ID1, char ID2, typename... Is1, typename... Is2, typename Inverter>
		struct Expression<'a', dimensions, T, IndexPackType<FreeIndices...>,
			Expression<ID1, dimensions, T, Is1...>, Expression<ID2, dimensions, T, Is2...>, Inverter>
		{
			Expression<ID1, dimensions, T, Is1...> val1;
			Expression<ID2, dimensions, T, Is2...> val2;

			//generic expression requirements

			inline T getValue()
			{
				if constexpr (Inverter::value)
				{
					return val1.getValue() - val2.getValue();
				}
				else
				{
					return val1.getValue() + val2.getValue();
				}
			}

			template<typename Index>
			inline void setIndex(const size_t& indexValue)
			{
				val1.template setIndex<Index>(indexValue);
				val2.template setIndex<Index>(indexValue);
			}
		};

		//expression for scalar multiplication
		template<size_t dimensions, typename T, typename... FreeIndices,
			char ID, typename... Is, typename Inverter>
		struct Expression<'m', dimensions, T, IndexPackType<FreeIndices...>, Expression<ID, dimensions, T, Is...>, Inverter>
		{
			T multiplier;
			Expression<ID, dimensions, T, Is...> val;

			//generic expression requirements

			inline T getValue()
			{
				if constexpr (Inverter::value)
				{
					return val.getValue() / multiplier;
				}
				else
				{
					return multiplier * val.getValue();
				}
			}

			template<typename Index>
			inline void setIndex(const size_t& indexValue)
			{
				val.template setIndex<Index>(indexValue);
			}
		};

		//template dependencies
		template<size_t dimensions, typename T, typename... FreeIndices1, typename... FreeIndices2,
			char ID1, char ID2, typename... Is1, typename... Is2,
			typename = std::enable_if_t<Template_Equal_Packs<IndexPackType<FreeIndices1...>, IndexPackType<FreeIndices2...>>::value>>
		//return type
		Expression<'a', dimensions, T, IndexPackType<FreeIndices1...>,
			Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...>,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...>, InverseType<false>>
		//operation
		operator+(Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...> left,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...> right)
		{
			return {left, right};
		}

		//template dependencies
		template<size_t dimensions, typename T, typename... FreeIndices1, typename... FreeIndices2,
			char ID1, char ID2, typename... Is1, typename... Is2,
			typename = std::enable_if_t<Template_Equal_Packs<IndexPackType<FreeIndices1...>, IndexPackType<FreeIndices2...>>::value>>
		//return type
		Expression<'a', dimensions, T, IndexPackType<FreeIndices1...>,
			Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...>,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...>, InverseType<true>>
		//operation
		operator-(Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...> left,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...> right)
		{
			return {left, right};
		}

		//template dependencies
		template<size_t dimensions, typename T, char ID1, char ID2,
			typename... FreeIndices1, typename... FreeIndices2, typename... Is1, typename... Is2>
		//return type
		Expression<'m', dimensions, T, typename Template_Remove_Repeats<FreeIndices1..., FreeIndices2...>::T,
			Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...>,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...>,
			typename Template_Get_Repeats<FreeIndices1..., FreeIndices2...>::T, InverseType<false>>
		//operation
		operator*(Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...> left,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...> right)
		{
			return {left, right};
		}

		//template dependencies
		template<size_t dimensions, typename T, char ID1, char ID2,
			typename... FreeIndices1, typename... FreeIndices2, typename... Is1, typename... Is2>
		//return type
		Expression<'m', dimensions, T, typename Template_Remove_Repeats<FreeIndices1..., FreeIndices2...>::T,
			Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...>,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...>,
			typename Template_Get_Repeats<FreeIndices1..., FreeIndices2...>::T, InverseType<true>>
		//operation
		operator/(Expression<ID1, dimensions, T, IndexPackType<FreeIndices1...>, Is1...> left,
			Expression<ID2, dimensions, T, IndexPackType<FreeIndices2...>, Is2...> right)
		{
			return {left, right};
		}

		//template dependencies
		template<size_t dimensions, typename T, typename... FreeIndices, char ID, typename... Is>
		//return type
		Expression<'m', dimensions, T, IndexPackType<FreeIndices...>,
			Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, Is...>, InverseType<false>>
		//operation
		operator*(T left, Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, Is...> right)
		{
			return {left, right};
		}

		//template dependencies
		template<size_t dimensions, typename T, typename... FreeIndices, char ID, typename... Is>
		//return type
		Expression<'m', dimensions, T, IndexPackType<FreeIndices...>,
			Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, Is...>, InverseType<false>>
		//operation
		operator*(Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, Is...> left, T right)
		{
			return {right, left};
		}

		//template dependencies
		template<size_t dimensions, typename T, typename... FreeIndices, char ID, typename... Is>
		//return type
		Expression<'m', dimensions, T, IndexPackType<FreeIndices...>,
			Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, Is...>, InverseType<true>>
		//operation
		operator/(Expression<ID, dimensions, T, IndexPackType<FreeIndices...>, Is...> left, T right)
		{
			return {right, left};
		}
	}

	//type differentiator for tensor indices
	template<char indexIdentifier>
	struct Index{static constexpr char value = indexIdentifier;};

	//non-indexed generic tensor type, serves as storage location for tensor data
	template<size_t dimensions, size_t rank, typename T = double,
		typename = std::enable_if_t<std::greater<size_t>()(dimensions, 1)>>
	// template<size_t dimensions, size_t rank, typename T>
	class Tensor
	{
		typedef Tensor<dimensions, rank, T> SelfType;
		T data[Template_Power<dimensions, rank>::value];
	public:
		Tensor()
		{
			// if constexpr (rank == 0) std::cout << "default constructing" << std::endl;
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] = T();
			}
		}

		Tensor(std::vector<T> inputValues)
		{
			std::copy(inputValues.begin(), inputValues.end(), data);
		}
		Tensor(T* inputValues)
		{
			std::copy(inputValues, inputValues + Template_Power<dimensions, rank>::value, data);
		}

		template<typename... IndexIdentifiers>
		auto operator()(IndexIdentifiers... indices) const//make single Expression
		{
			return Expression<'s', dimensions, T,
				typename Template_Remove_Repeats<IndexIdentifiers...>::T,
				IndexedTensor<rank, dimensions, T, IndexIdentifiers...>,
				typename Template_Get_Repeats<IndexIdentifiers...>::T>((T*)data);
		}

		template<typename S, typename = std::enable_if_t<!std::is_same<T, double>::value && std::is_same<T, S>::value>>
		SelfType& operator*=(S other)
		{
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] *= other;
			}
			return *this;
		}

		template<typename S, typename = std::enable_if_t<!std::is_same<T, double>::value && std::is_same<T, S>::value>>
		SelfType& operator/=(S other)
		{
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] /= other;
			}
			return *this;
		}

		SelfType& operator*=(double other)
		{
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] *= other;
			}
			return *this;
		}

		SelfType& operator/=(double other)
		{
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] /= other;
			}
			return *this;
		}

		SelfType& operator+=(const SelfType& other)
		{
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] += other.data[i];
			}
			return *this;
		}

		SelfType& operator-=(const SelfType& other)
		{
			for (size_t i = 0; i < Template_Power<dimensions, rank>::value; ++i)
			{
				data[i] -= other.data[i];
			}
			return *this;
		}

		T* getData()
		{
			return data;
		}

		const T* getData() const
		{
			return data;
		}

		std::vector<T> getDataCopy() const
		{
			return std::vector<T>(data, data + Template_Power<dimensions, rank>::value);
		}
	};

	template<size_t dimensions, typename T>
	void streamVector(std::ostream& os, const T* data)
	{
		os << "<" << data[0];
		for (size_t i = 1; i < dimensions; ++i)
		{
			os << ", " << data[i];
		}
		os << ">";
	}

	template<size_t dimensions, typename T>
	void streamMatrix(std::ostream& os, const T* data)
	{
		os << "[";
		streamVector<dimensions, T>(os, data);
		for (size_t i = 1; i < dimensions; ++i)
		{
			os << "\n ";
			streamVector<dimensions, T>(os, data + i * dimensions);
		}
		os << "]";
	}

	template<size_t dimensions, size_t rankRemaining, typename T>
	void streamSubTensor(std::ostream& os, const T* data, std::string head)
	{
		if constexpr (rankRemaining == 2)
		{
			os << head << "):\n";
			streamMatrix(os, data);
		}
		else
		{
			for (size_t i = 0; i < dimensions; ++i)
			{
				streamSubTensor<dimensions, rankRemaining - 1, T>
					(os, data + i * Template_Power<dimensions, rankRemaining - 1>::value,
						head + ", " + std::to_string(i));
			}
		}
	}

	template<size_t dimensions, size_t rank, typename T>
	std::ostream& operator<<(std::ostream& os, const Tensor<dimensions, rank, T>& thing)
	{
		os << "Rank " << rank << " " << dimensions << "D Tensor:";//usually 17 characters
		if constexpr (rank == 0)
		{
			os << " " << thing.getData()[0];
		}
		else if constexpr (rank == 1)
		{
			os << ' ';
			streamVector<dimensions, T>(os, thing.getData());
		}
		else if constexpr (rank == 2)
		{
			os << '\n';
			streamMatrix<dimensions, T>(os, thing.getData());
		}
		else
		{
			os << "\n";
			for (size_t i = 0; i < dimensions; ++i)
			{
				streamSubTensor<dimensions, rank, T, rank - 1>
					(os, thing.getData() + i * Template_Power<dimensions, rank - 1>::value,
						"Sub-matrix (" + std::to_string(i));
			}
		}
		return os;
	}

	template<size_t dimensions, size_t rank, typename T>
	Tensor<dimensions, rank, T> operator+(Tensor<dimensions, rank, T> left, const Tensor<dimensions, rank, T>& right)
	{
		return left += right;
	}

	template<size_t dimensions, size_t rank, typename T>
	Tensor<dimensions, rank, T> operator-(Tensor<dimensions, rank, T> left, const Tensor<dimensions, rank, T>& right)
	{
		return left -= right;
	}

	template<size_t dimensions, size_t rank, typename T, std::enable_if_t<!std::is_same<T, double>::value>>
	Tensor<dimensions, rank, T> operator*(Tensor<dimensions, rank, T> left, const T& right)
	{
		return left *= right;
	}

	template<size_t dimensions, size_t rank, typename T, std::enable_if_t<!std::is_same<T, double>::value>>
	Tensor<dimensions, rank, T> operator*(const T& left, Tensor<dimensions, rank, T> right)
	{
		return right *= left;
	}

	template<size_t dimensions, size_t rank, typename T, std::enable_if_t<!std::is_same<T, double>::value>>
	Tensor<dimensions, rank, T> operator/(Tensor<dimensions, rank, T> left, const T& right)
	{
		return left /= right;
	}

	template<size_t dimensions, size_t rank, typename T>
	Tensor<dimensions, rank, T> operator*(Tensor<dimensions, rank, T> left, const double& right)
	{
		return left *= right;
	}

	template<size_t dimensions, size_t rank, typename T>
	Tensor<dimensions, rank, T> operator*(const double& left, Tensor<dimensions, rank, T> right)
	{
		return right *= left;
	}

	template<size_t dimensions, size_t rank, typename T>
	Tensor<dimensions, rank, T> operator/(Tensor<dimensions, rank, T> left, const double& right)
	{
		return left /= right;
	}
}