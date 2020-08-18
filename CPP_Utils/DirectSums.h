namespace SimulationUtilities{

	template<typename... VectorTypes> class DirectSum;

	template<typename... VectorTypes>
	std::ostream& operator<<(std::ostream& os, const DirectSum<VectorTypes...>& thing);

	namespace
	{
		template<typename first, typename... VectorTypes>
		struct printHelper{
			static inline void formatPrint(std::ostream& os, const first& item, const VectorTypes&... others){
				os << item << ", ";
				printHelper<VectorTypes...>::formatPrint(os, others...);
			}
		};

		template<typename last>
		struct printHelper<last>{
			static inline void formatPrint(std::ostream& os, const last& item){
				os << item;
			}
		};
	}

	template<typename... VectorTypes>
	class DirectSum{
		typedef std::tuple<VectorTypes...> DataType;
		typedef DirectSum<VectorTypes...> SumType;
		DataType values;
		static inline auto seq(){
			return std::index_sequence_for<VectorTypes...>();
		}

		static inline void init(VectorTypes&... valsList){
			(void(valsList = VectorTypes()), ...);
		}
		template<size_t... Is>
		static inline void initTuple(DataType& input,
			std::index_sequence<Is...> seq)
		{
			init(std::get<Is>(input)...);
		}

		static inline void init(VectorTypes&... valsList, const VectorTypes&... initValues){
			(void(valsList = VectorTypes(initValues)), ...);
		}
		template<size_t... Is>
		static inline void initTuple(DataType& input, const VectorTypes&... initValues,
			std::index_sequence<Is...> seq)
		{
			init(std::get<Is>(input)..., initValues...);
		}

		template<size_t... Is>
		static inline void toStreamTuple(std::ostream& os, const DataType& thing,
			std::index_sequence<Is...> seq)
		{
			os << "<";
			printHelper<VectorTypes...>::formatPrint(os, std::get<Is>(thing)...);
			os << ">";
		}

		static inline void add(VectorTypes&... left, const VectorTypes&... right){
			(void(left += right),...);
		}
		template<size_t... Is>
		static inline void addTuple(DataType& left, const DataType& right,
			std::index_sequence<Is...> seq)
		{
			add(std::get<Is>(left)..., std::get<Is>(right)...);
		}

		static void sub(VectorTypes&... left, const VectorTypes&... right){
			(void(left -= right),...);
		}
		template<size_t... Is>
		static inline void subTuple(DataType& left, const DataType& right,
			std::index_sequence<Is...> seq)
		{
			sub(std::get<Is>(left)..., std::get<Is>(right)...);
		}

		static void mult(VectorTypes&... left, double right){
			(void(left *= right),...);
		}
		template<size_t... Is>
		static inline void multTuple(DataType& left, double right,
			std::index_sequence<Is...> seq)
		{
			mult(std::get<Is>(left)..., right);
		}

		static void div(VectorTypes&... left, double right){
			(void(left /= right),...);
		}
		template<size_t... Is>
		static inline void divTuple(DataType& left, double right,
			std::index_sequence<Is...> seq)
		{
			div(std::get<Is>(left)..., right);
		}

		static double dot(const VectorTypes&... left, const VectorTypes&... right){
			return ((left * right) + ...);
		}

		template<size_t... Is>
		static inline double dotTuple(const DataType& left, const DataType& right,
			std::index_sequence<Is...> seq)
		{
			return dot(std::get<Is>(left)..., std::get<Is>(right)...);
		}
	public:
		// DirectSum(){
		// 	initTuple(values, seq());
		// }

		// DirectSum(const VectorTypes&... initValues){
		// 	initTuple(values, initValues..., seq());
		// }
		DirectSum()
		:
			values(DataType())
		{}

		DirectSum(const VectorTypes&... initValues)
		:
			values(std::make_tuple(initValues...))
		{}

		template<size_t first, size_t... Is>
		friend struct Projection;

		friend std::ostream& operator<<<VectorTypes...>(std::ostream& os, const SumType& thing);
		
		SumType& operator+=(const SumType& other){
			addTuple(values, other.values, seq());
			return *this;
		}

		SumType& operator-=(const SumType& other){
			subTuple(values, other.values, seq());
			return *this;
		}

		SumType& operator*=(double other){
			multTuple(values, other, seq());
			return *this;
		}

		SumType& operator*=(float other){
			multTuple(values, other, seq());
			return *this;
		}

		SumType& operator*=(int other){
			multTuple(values, other, seq());
			return *this;
		}

		template<typename T>
		SumType& operator/=(T other){
			divTuple(values, other, seq());
			return *this;
		}

		double dotProduct(const SumType& other) const{
			return dotTuple(values, other.values, seq());
		}

		double defaultSquareMagnitude() const{
			return dotProduct(*this);
		}

		double defaultMagnitude() const
		{
			return std::sqrt(defaultSquareMagnitude());
		}

		DataType getData() const
		{
			return values;
		}
	};

	template<>
	class DirectSum<>{
		DirectSum<> operator+=(const DirectSum<>& other){
			return *this;
		}
		DirectSum<> operator-=(const DirectSum<>& other){
			return *this;
		}
		DirectSum<> operator*=(double other){
			return *this;
		}
		DirectSum<> operator/=(double other){
			return *this;
		}
	};

	template<typename... VectorTypes>
	std::ostream& operator<<(std::ostream& os, const DirectSum<VectorTypes...>& thing){
		DirectSum<VectorTypes...>::toStreamTuple(os, thing.values, DirectSum<VectorTypes...>::seq());
		return os;
	}

	template<typename... VectorTypes>
	DirectSum<VectorTypes...> operator+(DirectSum<VectorTypes...> left, const DirectSum<VectorTypes...>& right){
		return left += right;
	}

	template<typename... VectorTypes>
	DirectSum<VectorTypes...> operator-(DirectSum<VectorTypes...> left, const DirectSum<VectorTypes...>& right){
		return left -= right;
	}

	template<typename T, typename... VectorTypes>
	DirectSum<VectorTypes...> operator*(DirectSum<VectorTypes...> left, T right){
		return left *= right;
	}

	template<typename T, typename... VectorTypes>
	DirectSum<VectorTypes...> operator*(T left, DirectSum<VectorTypes...> right){
		return right *= left;
	}

	template<typename... VectorTypes>
	double operator*(DirectSum<VectorTypes...> left, DirectSum<VectorTypes...> right){
		return left.defualtDot(right);
	}

	template<typename T, typename... VectorTypes>
	DirectSum<VectorTypes...> operator/(DirectSum<VectorTypes...> left, T right){
		return left /= right;
	}

	template<size_t first, size_t... Is>
	struct Projection{
		template<typename... VectorTypes>
		static const auto& get(const DirectSum<VectorTypes...>& input){
			return Projection<Is...>::get(std::get<first>(input.values));
		}
		template<typename... VectorTypes>
		static auto& dynamicGet(DirectSum<VectorTypes...>& input){
			return Projection<Is...>::get(std::get<first>(input.values));
		}
	};

	template<size_t last>
	struct Projection<last>{
		template<typename... VectorTypes>
		static const auto& get(const DirectSum<VectorTypes...>& input){
			return std::get<last>(input.values);
		}
		template<typename... VectorTypes>
		static auto& dynamicGet(DirectSum<VectorTypes...>& input){
			return std::get<last>(input.values);
		}
	};

	template<size_t... Is, typename... VectorTypes>
	const auto& get(const DirectSum<VectorTypes...>& input){
		return Projection<Is...>::get(input);
	}

	template<size_t... Is, typename... VectorTypes>
	auto& getReference(DirectSum<VectorTypes...>& input){
		return Projection<Is...>::get(input);
	}

	template<size_t... Is, typename... VectorTypes>
	auto getCopy(const DirectSum<VectorTypes...>& input){
		return Projection<Is...>::get(input);
	}

	template<typename SetType, size_t... Is, typename... VectorTypes>
	void set(const DirectSum<VectorTypes...>& input, const SetType& value){
		Projection<Is...>::dynamicGet(input) = value;
	}

	template<typename... VectorTypes>
	double dotProduct(const DirectSum<VectorTypes...>& left, const DirectSum<VectorTypes...>& right)
	{
		return left.dotProduct(right);
	}

}