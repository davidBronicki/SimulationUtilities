
namespace SimulationUtilities{

	template<size_t dimensions, size_t rank, size_t divisions, typename T = double,
		typename = std::enable_if_t<dimensions != 0 && std::greater<size_t>()(divisions, 4)>>
	class TensorField;

	template<size_t dimensions, size_t divisions>
	using ScalarField = TensorField<dimensions, 0, divisions>;

	namespace
	{
		template<char ID, size_t dimensions, size_t divisions, typename T, typename... Is>
		struct TensorFieldExpression;

		//dynamic single expression type
		template<size_t dimensions, size_t divisions, size_t rank, typename T, typename... Is>
		struct TensorFieldExpression<'s', dimensions, divisions, T, Tensor<dimensions, rank, T>, Is...>
		{
			typedef TensorFieldExpression<'s', dimensions, divisions, T, Tensor<dimensions, rank, T>, Is...> SelfType;
			typedef Tensor<dimensions, rank, T> TensorType;

			static constexpr size_t tensorDataSize = Template_Power<divisions, dimensions>::value;

			std::shared_ptr<TensorType[tensorDataSize]> tensorData;

			TensorFieldExpression(const std::shared_ptr<TensorType[tensorDataSize]>& initData)
			:
				tensorData(initData)
			{}

			TensorFieldExpression(const SelfType& other)
			:
				tensorData(other.tensorData)
			{}

			template<char OtherID, typename... OtherIs>
			SelfType& operator=(TensorFieldExpression<OtherID, dimensions, divisions, OtherIs...>&& other)
			{
				for (size_t i = 0; i < tensorDataSize; ++i)
				{
					tensorData[i](Is()...) = other[i];
				}
				return *this;
			}

			SelfType& operator=(SelfType&& other)
			{
				for (size_t i = 0; i < tensorDataSize; ++i)
				{
					tensorData[i](Is()...) = other[i];
				}
				return *this;
			}

			template<char OtherID, typename... OtherIs>
			SelfType& operator+=(TensorFieldExpression<OtherID, dimensions, divisions, OtherIs...>&& other)
			{
				for (size_t i = 0; i < tensorDataSize; ++i)
				{
					tensorData[i](Is()...) += other[i];
				}
				return *this;
			}

			template<char OtherID, typename... OtherIs>
			SelfType& operator-=(TensorFieldExpression<OtherID, dimensions, divisions, OtherIs...>&& other)
			{
				for (size_t i = 0; i < tensorDataSize; ++i)
				{
					tensorData[i](Is()...) -= other[i];
				}
				return *this;
			}

			auto operator[](size_t index)
			{
				return tensorData[index](Is()...);
			}
		};

		template<size_t dimensions, size_t divisions, typename T, char ID1, char ID2, typename... Is1, typename... Is2, typename Inverter>
		struct TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...>,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...>, Inverter>
		{
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...> field1;
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...> field2;
			auto operator[](size_t index)
			{
				if constexpr (Inverter::value)
				{
					return field1[index] / field2[index];
				}
				else
				{
					return field1[index] * field2[index];
				}
			}
		};

		template<size_t dimensions, size_t divisions, typename T, char ID1, char ID2, typename... Is1, typename... Is2, typename Inverter>
		struct TensorFieldExpression<'a', dimensions, divisions, T, TensorFieldExpression<ID1, dimensions, divisions, T, Is1...>,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...>, Inverter>
		{
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...> field1;
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...> field2;
			auto operator[](size_t index)
			{
				if constexpr (Inverter::value)
				{
					return field1[index] - field2[index];
				}
				else
				{
					return field1[index] + field2[index];
				}
			}
		};

		template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is, typename Inverter>
		struct TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>, Inverter>
		{
			T multiplier;
			TensorFieldExpression<ID, dimensions, divisions, T, Is...> field;
			auto operator[](size_t index)
			{
				if constexpr (Inverter::value)
				{
					return field[index] / multiplier;
				}
				else
				{
					return field[index] * multiplier;
				}
			}
		};

		template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is,
			char OtherID, typename... OtherTs, typename Inverter>
		struct TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
			Expression<OtherID, dimensions, OtherTs...>, Inverter>
		{
			Expression<OtherID, dimensions, OtherTs...> multiplier;
			TensorFieldExpression<ID, dimensions, divisions, T, Is...> field;
			auto operator[](size_t index)
			{
				if constexpr (Inverter::value)
				{
					return field[index] / multiplier;
				}
				else
				{
					return field[index] * multiplier;
				}
			}
		};

		// template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is, typename Inverter>
		// struct TensorFieldExpression<'a', dimensions, divisions, T,
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...>, Inverter>
		// {
		// 	T addition;
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...> field;
		// 	auto operator[](size_t index)
		// 	{
		// 		if constexpr (Inverter::value)
		// 		{
		// 			return field[index] - addition;
		// 		}
		// 		else
		// 		{
		// 			return field[index] + addition;
		// 		}
		// 	}
		// };

		// template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is,
		// 	char OtherID, typename... OtherTs, typename Inverter>
		// struct TensorFieldExpression<'a', dimensions, divisions, T,
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
		// 	Expression<OtherID, dimensions, OtherTs...>, Inverter>
		// {
		// 	Expression<OtherID, dimensions, OtherTs...> addition;
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...> field;
		// 	auto operator[](size_t index)
		// 	{
		// 		if constexpr (Inverter::value)
		// 		{
		// 			return field[index] - addition;
		// 		}
		// 		else
		// 		{
		// 			return field[index] + addition;
		// 		}
		// 	}
		// };

		template<size_t dimensions, size_t divisions, typename T, char ID1, char ID2, typename... Is1, typename... Is2>
		TensorFieldExpression<'a', dimensions, divisions, T,
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...>,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...>, InverseType<false>>
		operator+(TensorFieldExpression<ID1, dimensions, divisions, T, Is1...> const& left,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...> const& right)
		{
			return {left, right};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID1, char ID2, typename... Is1, typename... Is2>
		TensorFieldExpression<'a', dimensions, divisions, T,
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...>,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...>, InverseType<true>>
		operator-(TensorFieldExpression<ID1, dimensions, divisions, T, Is1...> const& left,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...> const& right)
		{
			return {left, right};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID1, char ID2, typename... Is1, typename... Is2>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...>,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...>, InverseType<false>>
		operator*(TensorFieldExpression<ID1, dimensions, divisions, T, Is1...> const& left,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...> const& right)
		{
			return {left, right};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID1, char ID2, typename... Is1, typename... Is2>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID1, dimensions, divisions, T, Is1...>,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...>, InverseType<true>>
		operator/(TensorFieldExpression<ID1, dimensions, divisions, T, Is1...> const& left,
			TensorFieldExpression<ID2, dimensions, divisions, T, Is2...> const& right)
		{
			return {left, right};
		}

		// template<size_t dimensions, size_t divisions, typename T, char ID,
		// 	char OtherID, typename... OtherTs, typename... Is>
		// TensorFieldExpression<'a', dimensions, divisions, T,
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
		// 	Expression<OtherID, dimensions, OtherTs...>, InverseType<false>>
		// operator+(TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& left,
		// 	Expression<OtherID, dimensions, OtherTs...> const& right)
		// {
		// 	return {right, left};
		// }

		// template<size_t dimensions, size_t divisions, typename T, char ID,
		// 	char OtherID, typename... OtherTs, typename... Is>
		// TensorFieldExpression<'a', dimensions, divisions, T,
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
		// 	Expression<OtherID, dimensions, OtherTs...>, InverseType<false>>
		// operator+(Expression<OtherID, dimensions, OtherTs...> const& left,
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& right)
		// {
		// 	return {left, right};
		// }

		// template<size_t dimensions, size_t divisions, typename T, char ID,
		// 	char OtherID, typename... OtherTs, typename... Is>
		// TensorFieldExpression<'a', dimensions, divisions, T,
		// 	TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
		// 	Expression<OtherID, dimensions, OtherTs...>, InverseType<true>>
		// operator-(TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& left,
		// 	Expression<OtherID, dimensions, OtherTs...> const& right)
		// {
		// 	return {right, left};
		// }

		template<size_t dimensions, size_t divisions, typename T, char ID,
			char OtherID, typename... OtherTs, typename... Is>
		TensorFieldExpression<'a', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
			Expression<OtherID, dimensions, OtherTs...>, InverseType<false>>
		operator*(TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& left,
			Expression<OtherID, dimensions, OtherTs...> const& right)
		{
			return {right, left};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID,
			char OtherID, typename... OtherTs, typename... Is>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
			Expression<OtherID, dimensions, OtherTs...>, InverseType<false>>
		operator*(Expression<OtherID, dimensions, OtherTs...> const& left,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& right)
		{
			return {left, right};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID,
			char OtherID, typename... OtherTs, typename... Is>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>,
			Expression<OtherID, dimensions, OtherTs...>, InverseType<true>>
		operator/(TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& left,
			Expression<OtherID, dimensions, OtherTs...> const& right)
		{
			return {right, left};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>, InverseType<false>>
		operator*(TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& left, T const& right)
		{
			return {right, left};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>, InverseType<false>>
		operator*(T const& left, TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& right)
		{
			return {left, right};
		}

		template<size_t dimensions, size_t divisions, typename T, char ID, typename... Is>
		TensorFieldExpression<'m', dimensions, divisions, T,
			TensorFieldExpression<ID, dimensions, divisions, T, Is...>, InverseType<true>>
		operator/(TensorFieldExpression<ID, dimensions, divisions, T, Is...> const& left, T const& right)
		{
			return {right, left};
		}
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	class TensorField<dimensions, rank, divisions, T>
	{
		static constexpr size_t tensorDataSize = Template_Power<divisions, dimensions>::value;

		typedef TensorField<dimensions, rank, divisions, T> SelfType;
		typedef Tensor<dimensions, rank, T> TensorType;
		std::shared_ptr<TensorType[tensorDataSize]> tensorData;
	public:
		TensorField()
		:
			tensorData(new TensorType[tensorDataSize])
		{
			for (size_t i = 0; i < tensorDataSize; ++i)
			{
				tensorData[i] = TensorType();
			}
		}
		TensorField(const std::vector<TensorType>& input)
		:
			tensorData(new TensorType[tensorDataSize])
		{
			std::copy(input.begin(), input.end(), tensorData);
		}

		TensorField(SelfType&& other) = default;
		// :
		// 	tensorData(other.tensorData)
		// {}

		TensorField(const SelfType& other)
		:
			tensorData(new TensorType[tensorDataSize])
		{
			std::copy(other.tensorData.get(), other.tensorData.get() + tensorDataSize, tensorData.get());
		}

		SelfType& operator=(const SelfType& other)
		{
			tensorData = std::shared_ptr<TensorType[tensorDataSize]>(new TensorType[tensorDataSize]);
			std::copy(other.tensorData.get(), other.tensorData.get() + tensorDataSize, tensorData.get());
			return *this;
		}
		SelfType& operator=(SelfType&& other) = default;

		template<typename... IndexIdentifiers>
		auto operator()(IndexIdentifiers... indices) const//make constant tensorData Expression
		{
			return TensorFieldExpression<'s', dimensions, divisions, T, TensorType, IndexIdentifiers...>(tensorData);
		}

		SelfType& operator+=(const SelfType& other)
		{
			for (size_t i = 0; i < tensorDataSize; ++i)
			{
				tensorData[i] += other.tensorData[i];
			}
			return *this;
		}
		SelfType& operator-=(const SelfType& other)
		{
			for (size_t i = 0; i < tensorDataSize; ++i)
			{
				tensorData[i] -= other.tensorData[i];
			}
			return *this;
		}
		SelfType& operator*=(double other){
			for (size_t i = 0; i < tensorDataSize; ++i)
			{
				tensorData[i] *= other;
			}
			return *this;
		}
		SelfType& operator/=(double other){
			for (size_t i = 0; i < tensorDataSize; ++i)
			{
				tensorData[i] /= other;
			}
			return *this;
		}

		TensorType& operator[](size_t index)
		{
			return tensorData[index];
		}

		const TensorType& operator[](size_t index) const
		{
			return tensorData[index];
		}

		template<size_t dimension>
		static inline size_t stepSize()
		{
			return Template_Power<divisions, dimensions - dimension - 1>::value;
		}

		const TensorType* begin() const
		{
			return tensorData.get();
		}

		const TensorType* end() const
		{
			return tensorData.get() + tensorDataSize;
		}
	};

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	auto operator+(TensorField<dimensions, rank, divisions, T> left,
		const TensorField<dimensions, rank, divisions, T>& right)
	{
		return left += right;
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	auto operator-(TensorField<dimensions, rank, divisions, T> left,
		const TensorField<dimensions, rank, divisions, T>& right)
	{
		return left -= right;
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	auto operator*(TensorField<dimensions, rank, divisions, T> left, const T& right)
	{
		return left *= right;
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	auto operator*(const T& left, TensorField<dimensions, rank, divisions, T> right)
	{
		return right *= left;
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	auto operator/(TensorField<dimensions, rank, divisions, T> left, const T& right)
	{
		return left /= right;
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	TensorField<dimensions, rank + 1, divisions, T> gradient_ignoreBoundary(
		const TensorField<dimensions, rank, divisions, T>& input, double dx)
	{
		//perform a fourth order gradient on a tensor field, producing a rank n+1 tensor field
		//where the last index (though no indices are used here) is the derivative direction.

		typedef Tensor<dimensions, rank, T> TensorType;
		typedef Tensor<dimensions, rank + 1, T> NewTensorType;
		typedef TensorField<dimensions, rank, divisions, T> InputType;
		typedef TensorField<dimensions, rank + 1, divisions, T> OutputType;

		OutputType output;
		NewTensorType* outputData = (NewTensorType*)(output.begin());
		const TensorType* inputData = input.begin();

		size_t incr = 1;

		for (size_t dim = dimensions; dim > 0; --dim, incr *= divisions)
		{
			// size_t tensorDataOffset = (dim - 1) * Template_Power<dimensions, rank>::value;
			size_t tensorDataOffset = dim - 1;

			for (size_t i = 0; i < Template_Power<divisions, dimensions>::value; ++i)
			{
				size_t dimensionPosition = i / incr % divisions;

				if (dimensionPosition > 1)
				{
					if (dimensionPosition < divisions - 2)
					{
						//standard 4th order algorithm
						const TensorType* neg2 = inputData + i - incr * 2;
						const TensorType* neg1 = inputData + i - incr;
						const TensorType* pos1 = inputData + i + incr;
						const TensorType* pos2 = inputData + i + incr * 2;
						*((TensorType*)(outputData + i) + tensorDataOffset) = (8 * (*pos1 - *neg1) - *pos2 + *neg2)/(12 * dx);
					}
					else
					{
						//close to end
						if (dimensionPosition == divisions - 1)
						{
							//at end
							const TensorType* neg4 = inputData + i - incr * 4;
							const TensorType* neg3 = inputData + i - incr * 3;
							const TensorType* neg2 = inputData + i - incr * 2;
							const TensorType* neg1 = inputData + i - incr;
							const TensorType* p0 = inputData + i;
							*((TensorType*)(outputData + i) + tensorDataOffset)
								= (25 * *p0 - 48 * *neg1 + 36 * *neg2 - 16 * *neg3 + 3 * *neg4)/(12 * dx);
						}
						else
						{
							//one from end
							const TensorType* neg3 = inputData + i - incr * 3;
							const TensorType* neg2 = inputData + i - incr * 2;
							const TensorType* neg1 = inputData + i - incr;
							const TensorType* p0 = inputData + i;
							const TensorType* pos1 = inputData + i + incr;
							*((TensorType*)(outputData + i) + tensorDataOffset)
								= (3 * *pos1 - *neg3 + 6 * *neg2 + 10 * *p0 - 18 * *neg1)/(12 * dx);
						}
					}
				}
				else
				{
					//close to start
					if (dimensionPosition == 0)
					{
						//at start
						const TensorType* pos4 = inputData + i + incr * 4;
						const TensorType* pos3 = inputData + i + incr * 3;
						const TensorType* pos2 = inputData + i + incr * 2;
						const TensorType* pos1 = inputData + i + incr;
						const TensorType* p0 = inputData + i;
						*((TensorType*)(outputData + i) + tensorDataOffset)
							= (-25 * *p0 + 48 * *pos1 - 36 * *pos2 + 16 * *pos3 - 3 * *pos4)/(12 * dx);
					}
					else
					{
						//one from start
						const TensorType* pos3 = inputData + i + incr * 3;
						const TensorType* pos2 = inputData + i + incr * 2;
						const TensorType* pos1 = inputData + i + incr;
						const TensorType* p0 = inputData + i;
						const TensorType* neg1 = inputData + i - incr;
						*((TensorType*)(outputData + i) + tensorDataOffset)
							= (-3 * *neg1 + *pos3 - 6 * *pos2 - 10 * *p0 + 18 * *pos1)/(12 * dx);
					}
				}
			}
		}

		return output;
	}

	template<size_t dimensions, size_t rank, size_t divisions, typename T>
	TensorField<dimensions, rank + 1, divisions, T> gradient_periodicBoundary(
		const TensorField<dimensions, rank, divisions, T>& input, double dx)
	{
		//perform a fourth order gradient on a tensor field, producing a rank n+1 tensor field
		//where the last index (though no indices are used here) is the derivative direction.
		//the boundaries use the opposite side to create periodic boundary conditions

		typedef Tensor<dimensions, rank, T> TensorType;
		typedef Tensor<dimensions, rank + 1, T> NewTensorType;
		typedef TensorField<dimensions, rank, divisions, T> InputType;
		typedef TensorField<dimensions, rank + 1, divisions, T> OutputType;

		OutputType output;
		NewTensorType* outputData = (NewTensorType*)(output.begin());
		const TensorType* inputData = input.begin();

		size_t incr = 1;

		for (size_t dim = dimensions; dim > 0; --dim, incr *= divisions)
		{
			// size_t tensorDataOffset = (dim - 1) * Template_Power<dimensions, rank>::value;
			size_t tensorDataOffset = dim - 1;

			for (size_t i = 0; i < Template_Power<divisions, dimensions>::value; ++i)
			{
				size_t dimensionPosition = i / incr % divisions;

				if (dimensionPosition > 1 && dimensionPosition < divisions - 2)
				{
					//standard 4th order algorithm
					const TensorType* neg2 = inputData + i - incr * 2;
					const TensorType* neg1 = inputData + i - incr;
					const TensorType* pos1 = inputData + i + incr;
					const TensorType* pos2 = inputData + i + incr * 2;
					*((TensorType*)(outputData + i) + tensorDataOffset) = (8 * (*pos1 - *neg1) - *pos2 + *neg2)/(12 * dx);
				}
				else
				{
					const TensorType* neg2 = inputData + i + incr * ((dimensionPosition - 2 + divisions) % divisions - dimensionPosition);
					const TensorType* neg1 = inputData + i + incr * ((dimensionPosition - 1 + divisions) % divisions - dimensionPosition);
					const TensorType* pos1 = inputData + i + incr * ((dimensionPosition + 1) % divisions - dimensionPosition);
					const TensorType* pos2 = inputData + i + incr * ((dimensionPosition + 2) % divisions - dimensionPosition);
					*((TensorType*)(outputData + i) + tensorDataOffset) = (8 * (*pos1 - *neg1) - *pos2 + *neg2)/(12 * dx);
				}
			}
		}

		return output;
	}

}
